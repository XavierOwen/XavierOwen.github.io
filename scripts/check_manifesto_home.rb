#!/usr/bin/env ruby
# frozen_string_literal: true

require "pathname"

ROOT = Pathname(File.expand_path("..", __dir__))
ROUTES = %w[research-teaching notes-writing faith-spirituality projects-creation].freeze

def read_page(relative_path)
  path = ROOT.join("_site", relative_path)
  raise "missing rendered page: #{relative_path}" unless path.file?

  path.read(encoding: "UTF-8")
end

def assert_order!(html, selectors, label)
  positions = selectors.map { |selector| html.index(selector) || raise("#{label}: missing #{selector}") }
  raise "#{label}: hierarchy is out of order" unless positions == positions.sort
end

pages = {
  "index.html" => { language: "zh", prefix: "", alternate: "/en/" },
  "en/index.html" => { language: "en", prefix: "/en", alternate: "/" }
}

pages.each do |relative_path, expectation|
  html = read_page(relative_path)
  language = expectation.fetch(:language)
  assert_order!(html, ["manifesto-home__hero", "manifesto-home__paths", "manifesto-home__identity"], relative_path)
  raise "#{relative_path}: expected four equal reader-path entries" unless html.scan(/class="manifesto-home__path"/).length == 4
  raise "#{relative_path}: incorrect document language" unless html.include?(%(<html lang="#{language}"))
  raise "#{relative_path}: missing persisted-language route context" unless html.include?(%(data-language-context="route"))
  raise "#{relative_path}: missing compact Chinese control" unless html.include?(%(<a class="language-choice")) && html.include?(">中</a>")
  raise "#{relative_path}: missing compact English control" unless html.include?(">EN</a>")

  ROUTES.each do |route|
    href = language == "zh" ? "/paths/#{route}/" : "/en/paths/#{route}/"
    raise "#{relative_path}: missing #{route} entry" unless html.include?(%(href="#{href}"))
  end

  raise "#{relative_path}: professional identity must link to About" unless html.include?(language == "zh" ? %(href="/about/">) : %(href="/en/about/">))
  raise "#{relative_path}: professional identity must link to CV" unless html.include?(%(href="/cv/">))
  raise "#{relative_path}: collection tags leaked into primary navigation" if html.include?(%(<a href="/tags/"))
  raise "#{relative_path}: missing reciprocal home hreflang" unless html.include?(%(rel="alternate" hreflang="#{language == 'zh' ? 'en' : 'zh'}" href="https://XavierOwen.github.io#{expectation.fetch(:alternate)}"))
end

%w[about/index.html en/about/index.html].each { |path| read_page(path) }
raise "throwaway bilingual-home prototype is still published" if ROOT.join("_site/prototype/bilingual-home/index.html").exist?

css = ROOT.join("_site/assets/css/main.css").read(encoding: "UTF-8")
raise "language controls do not have fixed circular geometry" unless css.include?("width:2.65rem") && css.include?("height:2.65rem") && css.include?("place-items:center")

puts "Manifesto home static checks: passed in Chinese and English."
