#!/usr/bin/env ruby
# frozen_string_literal: true

require "pathname"
require "set"

ROOT = Pathname.new(File.expand_path("..", __dir__))
SITE = ROOT.join("_site")
MAIN_CSS = SITE.join("assets/css/main.css")

failures = []
asset_references = 0
asset_versions = Set.new

Dir.glob(SITE.join("**/*.html")).sort.each do |path|
  html = File.read(path)
  relative_path = Pathname.new(path).relative_path_from(SITE)

  html.scan(/<(?:link|script)\b[^>]*>/i).each do |tag|
    next if tag.start_with?("<link") && !tag.match?(/\brel="stylesheet"/i)

    asset_url = tag[/\b(?:href|src)="([^"]+)"/, 1]
    next unless asset_url&.match?(%r{/assets/(?:css|js)/})

    asset_references += 1
    version = asset_url[/[?&]v=([A-Za-z0-9._-]+)/, 1]

    if version
      asset_versions << version
    else
      failures << "#{relative_path}: unversioned static asset #{asset_url}"
    end
  end
end

failures << "No CSS or JavaScript asset references were found in _site HTML." if asset_references.zero?
failures << "Static assets use more than one build version: #{asset_versions.to_a.sort.join(', ')}" if asset_versions.length > 1

unless MAIN_CSS.file?
  failures << "Compiled stylesheet is missing: #{MAIN_CSS.relative_path_from(ROOT)}"
else
  css = MAIN_CSS.read
  %w[.manifesto-home__frame .reader-path__item .language-choice].each do |selector|
    failures << "Compiled main.css is missing #{selector}." unless css.include?(selector)
  end
end

if failures.any?
  warn "Static-asset checks failed:"
  failures.first(20).each { |failure| warn "- #{failure}" }
  warn "- ... #{failures.length - 20} more failure(s)" if failures.length > 20
  exit 1
end

puts "Static-asset checks passed for #{asset_references} versioned references (version #{asset_versions.first})."
